/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * 本文件将水果新鲜度检测wk模型部署到板端，通过NNIE硬件加速进行推理。
 *
 * This file deploys the fruit freshness detection wk model to the board,
 * and performs inference through NNIE hardware acceleration.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <sys/time.h>  // 添加时间处理头文件

#include "sample_comm_nnie.h"
#include "sample_media_ai.h"
#include "ai_infer_process.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "posix_help.h"
#include "base_interface.h"
#include "osd_img.h"
#include "cnn_trash_classify.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define MODEL_FILE_TRASH    "/userdata/models/cnn_trash_classify/resnet_inst.wk" // Open source model conversion
#define SCORE_MAX           4096
#define DETECT_OBJ_MAX      32
#define RET_NUM_MAX         4
#define THRESH_MIN          30
#define DETECTION_INTERVAL_SEC 10  // 10秒检测间隔

#define FRM_WIDTH           256
#define FRM_HEIGHT          256
#define TXT_BEGX            20
#define TXT_BEGY            20

#define MULTIPLE_OF_EXPANSION 100
#define BUFFER_SIZE           16
#define MIN_OF_BOX            16
#define MAX_OF_BOX            240

static OsdSet* g_osdsFruit = NULL;
static HI_S32 g_osd0Fruit = -1;

// 水果类别定义
static const HI_CHAR *fruitTypes[10] = {
    "fresh apple", "fresh banana", "fresh mango", "fresh orange", "fresh strawberry",
    "rotten apple", "rotten banana", "rotten mango", "rotten orange", "rotten strawberry"
};

/*
 * 加载水果新鲜度检测wk模型
 */
HI_S32 FruitFreshnessLoadModel(uintptr_t* model, OsdSet* osds)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    ret = OsdLibInit();
    HI_ASSERT(ret == HI_SUCCESS);

    g_osdsFruit = osds;
    HI_ASSERT(g_osdsFruit);
    g_osd0Fruit = OsdsCreateRgn(g_osdsFruit);
    HI_ASSERT(g_osd0Fruit >= 0);

    ret = CnnCreate(&self, MODEL_FILE_FRUIT);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    SAMPLE_PRT("load fruit freshness model, ret:%d\n", ret);

    return ret;
}

/*
 * 卸载水果新鲜度检测模型
 */
HI_S32 FruitFreshnessUnloadModel(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    SAMPLE_PRT("unload fruit freshness model success\n");
    OsdsClear(g_osdsFruit);

    return HI_SUCCESS;
}

/*
 * 根据推理结果进行业务处理
 */
static HI_S32 FruitFreshnessFlag(const RecogNumInfo items[], HI_S32 itemNum, HI_CHAR* buf, HI_S32 size)
{
    HI_S32 offset = 0;
    HI_BOOL foundValidResult = HI_FALSE;

    for (HI_U32 i = 0; i < itemNum; i++) {
        const RecogNumInfo *item = &items[i];
        HI_FLOAT confidence = (HI_FLOAT)item->score / SCORE_MAX;
        
        if (confidence * 100 < THRESH_MIN) {
            continue;
        }
        
        if (item->num >= 0 && item->num <= 9) {
            offset += snprintf_s(buf + offset, size - offset, size - offset - 1,
                "%s %.2f", fruitTypes[item->num], confidence);
            foundValidResult = HI_TRUE;
            break;
        }
    }

    if (!foundValidResult) {
        offset += snprintf_s(buf + offset, size - offset, size - offset - 1,
            "No fruit detected");
    }
    
    return HI_SUCCESS;
}

/*
 * 水果新鲜度检测主函数（每10秒执行一次）
 */
HI_S32 FruitFreshnessCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *resFrm)
{
    static struct timeval lastDetectTime = {0};  // 上次检测时间
    struct timeval currentTime;
    double elapsedSec = 0.0;
    
    // 获取当前时间
    gettimeofday(&currentTime, NULL);
    
    // 计算经过的时间
    if (lastDetectTime.tv_sec != 0) {
        elapsedSec = (currentTime.tv_sec - lastDetectTime.tv_sec) +
                     (currentTime.tv_usec - lastDetectTime.tv_usec) / 1000000.0;
    }
    
    // 如果距离上次检测不足10秒，跳过本次检测
    if (elapsedSec > 0 && elapsedSec < DETECTION_INTERVAL_SEC) {
        return HI_SUCCESS;
    }
    
    // 更新检测时间
    lastDetectTime = currentTime;
    
    SAMPLE_PRT("Starting fruit freshness detection (every %d seconds)\n", DETECTION_INTERVAL_SEC);
    
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    IVE_IMAGE_S img;
    RectBox cnnBoxs[DETECT_OBJ_MAX] = {0};
    VIDEO_FRAME_INFO_S resizeFrm;
    static HI_CHAR prevOsd[NORM_BUF_SIZE] = "";
    HI_CHAR osdBuf[NORM_BUF_SIZE] = "";
    RecogNumInfo resBuf[RET_NUM_MAX] = {0};
    HI_S32 resLen = 0;
    HI_S32 ret;
    IVE_IMAGE_S imgIn;

    // 设置检测区域（中心区域）
    cnnBoxs[0].xmin = MIN_OF_BOX;
    cnnBoxs[0].xmax = MAX_OF_BOX;
    cnnBoxs[0].ymin = MIN_OF_BOX;
    cnnBoxs[0].ymax = MAX_OF_BOX;

    // 调整帧大小以适应模型输入
    ret = MppFrmResize(srcFrm, &resizeFrm, FRM_WIDTH, FRM_HEIGHT);
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "for resize FAIL, ret=%x\n", ret);

    // 转换为IVE图像格式
    ret = FrmToOrigImg(&resizeFrm, &img);
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "for Frm2Img FAIL, ret=%x\n", ret);

    // 裁剪图像到检测区域
    ret = ImgYuvCrop(&img, &imgIn, &cnnBoxs[0]);
    SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "ImgYuvCrop FAIL, ret=%x\n", ret);

    // 使用NNIE进行推理
    ret = CnnCalImg(self, &imgIn, resBuf, sizeof(resBuf) / sizeof((resBuf)[0]), &resLen);
    SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "cnn cal FAIL, ret=%x\n", ret);

    HI_ASSERT(resLen <= sizeof(resBuf) / sizeof(resBuf[0]));
    
    // 处理推理结果
    ret = FruitFreshnessFlag(resBuf, resLen, osdBuf, sizeof(osdBuf));
    SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "FruitFreshnessFlag cal FAIL, ret=%x\n", ret);

    // 当结果发生变化时更新OSD显示
    if (strcmp(osdBuf, prevOsd) != 0) {
        HiStrxfrm(prevOsd, osdBuf, sizeof(prevOsd));
        HI_OSD_ATTR_S rgn;
        TxtRgnInit(&rgn, osdBuf, TXT_BEGX, TXT_BEGY, ARGB1555_YELLOW2);
        OsdsSetRgn(g_osdsFruit, g_osd0Fruit, &rgn);
        
        // 向VPSS发送数据
        ret = HI_MPI_VPSS_SendFrame(0, 0, srcFrm, 0);
        if (ret != HI_SUCCESS) {
            SAMPLE_PRT("Error(%#x), HI_MPI_VPSS_SendFrame failed!\n", ret);
        }
    }

    // 释放资源
    IveImgDestroy(&imgIn);
    MppFrmDestroy(&resizeFrm);

    return ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */