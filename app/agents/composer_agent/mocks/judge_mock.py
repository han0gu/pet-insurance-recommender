def get_mock_validation_result():
    """Judge Agent가 방금 성공적으로 만들어낸 검증 결과 (Mock)"""
    return {
        'selected_policies': [
            {
                'product_name': '[B사 시니어 케어]',
                'suitability_score': 85,
                'reason': '10세(15세 이하 가입 가능), 슬개골 탈구 및 심장판막증 보장 가능. 단 간질 보장 여부 미확인'
            },
            {
                'product_name': '[C사 실속 보험]',
                'suitability_score': 60,
                'reason': '전 연령 가입 가능하나 수술비 미보장으로 심장판막증/슬개골 탈구 치료에 제한적.'
            },
            {
                'product_name': '[A사 튼튼 펫보험]',
                'suitability_score': 20,
                'reason': '나이 제한(8세 이하) 및 슬개골 탈구 면책 조항으로 인해 부적합'
            }
        ],
        'review_summary': '최종 3개 상품 중 [B사 시니어 케어]가 가장 적합하나, 간질 보장 여부 추가 확인 필요.'
    }