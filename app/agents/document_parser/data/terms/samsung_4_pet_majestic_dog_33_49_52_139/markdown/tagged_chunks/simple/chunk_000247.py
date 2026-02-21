from langchain_core.documents import Document

chunk = Document(
    page_content=('- 니다.\n'
 '- ④ 보험료 납입이 연체중이라도 계약의 해지 전에 발생한 보험금 지급사유에 대하여 회\n'
 '- 사는 보상하며, 계약의 해지 전에 보험료 납입면제 사유가 발생한 경우 회사는 계약을\n'
 '- 해지하지 않고 차회 이후의 보험료 납입을 면제하여 드립니다.\n'
 '- ⑤ 회사가 제1항에 의한 납입최고(독촉) 등을 전자문서로 안내하고자 할 경우에는 계약자\n'
 '- 에게 서면, 전자서명법 제2조 제2호에 따른 전자서명으로 동의를 얻어 수신확인을 조\n'
 '- 건으로 전자문서를 송신하여야 하며, 계약자가 전자문서에 대하여 수신을 확인하기'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000247',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
