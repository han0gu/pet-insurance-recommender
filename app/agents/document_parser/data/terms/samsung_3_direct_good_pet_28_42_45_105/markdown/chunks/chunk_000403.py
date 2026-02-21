from langchain_core.documents import Document

chunk = Document(
    page_content=('사용된 연료를 포함합니다.\n'
 '장비용 등 가입동물의 사망 후에 소요된 비용, 각종 증명서류의 작성비용(운송비\n'
 '[핵연료물질에 의하여 오염된 물질]\n'
 '포함)\n'
 '원자핵 분열 생성물을 포함합니다.\n'
 '12. 펫호텔 비용 또는 위탁료, 산책료, 카운슬링 비용, 상담료, 지도료, 문제행동 교정\n'
 '6. 피보험자의 질병, 심신상실 또는 정신질환으로 인한 손해\n'
 '비용 및 이와 동종의 비용\n'
 '7. 최초계약의 보험계약일 이전에 이미 감염 또는 발병한 상해 및 질병\n'
 '13. 아래의 질병으로 인하여 발생한 손해는 보상하지 않습니다. 다만, 질병의 발생일'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
