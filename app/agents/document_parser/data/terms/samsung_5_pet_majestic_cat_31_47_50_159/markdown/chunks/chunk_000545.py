from langchain_core.documents import Document

chunk = Document(
    page_content=('- 「보험계약일」 이라 합니다)부터 그 날을 포함하여 30일이 지난 날의 다음날로 합니\n'
 '- 다. 다만, 상해를 직접적인 원인으로 수술을 받은 경우에는 보험계약일을 보장개시일(\n'
 '- 책임개시일)로 합니다. 이 경우 보험계약일은 이 추가특별약관의 제1회 보험료를 받은\n'
 '- 날로 합니다.\n'
 '<예시안내># 「반려묘 수술비(치과및 구강질환포함)(재가입형) 확대보장」 에 대한 보장개시일(책임개시일) 계\n'
 '산]![image](/image/placeholder)\n'
 '보험계약일 보장개시일(책임개시일)\n'
 '30일'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
