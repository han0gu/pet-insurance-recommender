from langchain_core.documents import Document

chunk = Document(
    page_content=('다는 사실을 미리 안내하고 동의를 받을 것\n'
 '2. 전자적 상품설명장치를 활용하여 안내한 납입최고(독촉) 등을 계약자가 모두 수신하\n'
 '고 이해하였음을 확인할 것\n'
 '3. 계약자가 질의를 하거나 추가적인 설명을 요청하는 등 전자적 상품설명장치의 활용\n'
 '을 중단할 것을 요구하는 경우, 회사는 전화 (음성녹음) 방법으로 전환하여 제1항에\n'
 '따른 납입최고(독촉) 등을 실시할 것4. 전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 있는 기능을 갖출 것'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000088',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
