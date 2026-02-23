from langchain_core.documents import Document

chunk = Document(
    page_content=('- 원없이 계속 입원중인 경우에는 반려견 위탁비용이 지급된 최종 입원일의 그 다음날\n'
 '- 을 퇴원일로 봅니다.\n'
 '![image](/image/placeholder)\n'
 '- Chart Title: <시안내>\n'
 '- Chart Type: bar\n'
 '|  | 보상 | 보상제외 |\n'
 '| --- | --- | --- |\n'
 '| 퉈원없이 계속 입원 | -182년 | -182년 |\n'
 '| 치고된 최종 입원이 | -173년 | -172년 |\n'
 '7 제1항의 경우 피보험자가 보장개시일(책임개시일) 이후 입원하여 치료를 받던 중 보험'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
