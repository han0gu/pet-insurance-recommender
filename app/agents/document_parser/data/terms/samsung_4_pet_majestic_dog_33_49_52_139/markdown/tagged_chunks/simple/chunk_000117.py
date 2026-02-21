from langchain_core.documents import Document

chunk = Document(
    page_content=('- 전자적 상품설명장치를 활용할 수 있습니다.\n'
 '- 1.계약자에게 전자적 상품설명장치를 활용하여 제1항에 따른 납입최고(독촉) 등을 한\n'
 '- 다는 사실을 미리 안내하고 동의를 받을 것\n'
 '- 2.전자적 상품설명장치를 활용하여 안내한 납입최고(독촉) 등을 계약자가 모두 수신하\n'
 '- 고 이해하였음을 확인할 것\n'
 '- 3.계약자가 질의를 하거나 추가적인 설명을 요청하는 등 전자적 상품설명장치의 활용\n'
 '- 을 중단할 것을 요구하는 경우, 회사는 전화 (음성녹음) 방법으로 전환하여 제1항에\n'
 '- 따른 납입최고(독촉) 등을 실시할 것'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000117',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
