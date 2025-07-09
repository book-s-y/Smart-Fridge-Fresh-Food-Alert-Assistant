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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <sys/time.h>

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

#define MODEL_FILE_TRASH    "/userdata/models/cnn_trash_classify/resnet_inst.wk"
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

static OsdSet* g_osdsTrash = NULL;
static HI_S32 g_osd0Trash = -1;

// 水果类别定义
static const HI_CHAR *fruitTypes[10] = {
    "fresh apple", "fresh banana", "fresh mango", "fresh orange", "fresh strawberry",
    "rotten apple", "rotten banana", "rotten mango", "rotten orange", "rotten strawberry"
};

/*
 * 加载垃圾分类wk模型
 */
HI_S32 CnnTrashClassifyLoadModel(uintptr_t* model, OsdSet* osds)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    ret = OsdLibInit();
    HI_ASSERT(ret == HI_SUCCESS);

    g_osdsTrash = osds;
    HI_ASSERT(g_osdsTrash);
    g_osd0Trash = OsdsCreateRgn(g_osdsTrash);
    HI_ASSERT(g_osd0Trash >= 0);

    ret = CnnCreate(&self, MODEL_FILE_TRASH);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    SAMPLE_PRT("load cnn trash classify model, ret:%d\n", ret);
    pthread_t streaming_thread;
    pthread_create(&streaming_thread, NULL, (void *)HiSteramingServer, NULL);
    return ret;
}

/*
 * 卸载垃圾分类wk模型
 */
HI_S32 CnnTrashClassifyUnloadModel(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    SAMPLE_PRT("unload trash classify model success\n");
    OsdsClear(g_osdsTrash);

    return HI_SUCCESS;
}

/*
 * 根据推理结果进行业务处理
 */
static HI_S32 CnnTrashClassifyFlag(const RecogNumInfo items[], HI_S32 itemNum, HI_CHAR* buf, HI_S32 size)
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
 * 水果新鲜度检测主函数
 */
HI_S32 CnnTrashClassifyCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *resFrm)
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
    
    // 处理推理结果（结果将存储在osdBuf中）
    ret = CnnTrashClassifyFlag(resBuf, resLen, osdBuf, sizeof(osdBuf));
    SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "FruitFreshnessFlag cal FAIL, ret=%x\n", ret);

    // 在终端输出结果（不显示在屏幕上）
    SAMPLE_PRT("Detection Result: %s\n", osdBuf);

    // 释放资源
    IveImgDestroy(&imgIn);
    MppFrmDestroy(&resizeFrm);
    
    SAMPLE_PRT("--- Detection completed. Next in %d seconds ---\n\n", DETECTION_INTERVAL_SEC);

    return ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */