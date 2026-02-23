from langchain_core.documents import Document

chunk = Document(
    page_content=('|  | 보상 | 보상제외 |\n'
 '| --- | --- | --- |\n'
 '| 틔원없이 계속 입원 | 0.06년 | 0.02년 |\n'
 '| 치고된 최종 입원일 | 0.08년 | 0.05년 |\n'
 '# 려견 위탁비용을 계속 보장합니다.⑧ 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회사는\n'
 '반려견 위탁비용의 전부 또는 일부를 지급하지 않습니다.# 제2조 (보험금 지급에 관한 세부규정)보험수익자와 회사가 제1조(보험금의 '
 '지급사유)의 보험금 지급사유에 대해 합의하지 못'),
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
