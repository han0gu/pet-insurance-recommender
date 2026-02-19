from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 보험료 납입이 연체중이라도 계약의 해지 전에 발생한 보험금 지급사유에 대하여 회 사는 보상하여 드립니다. ⑤ 회사가 제1항에 따른 '
 '납입최고(독촉) 등을 전자문서로 안내하고자 할 경우에는 계약자 에게 서면, 전자서명법 제2조 제2호에 따른 전자서명으로 동의를 얻어 수신 '
 '확인을 조건으로 전자문서를 송신하여야 하며, 계약자가 전자문서에 대하여 수신을 확인하기 전까지는 그 전자문서는 송신되지 않은 것으로 '
 '봅니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 43},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000132',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
