from langchain_core.documents import Document

chunk = Document(
    page_content=('[컴퓨터단층촬영(CT)]\n'
 'X선을 투과시켜 그 흡수차이를 컴퓨터로 재구성하여 신체의 단면영상을 얻거나 3차원적인 입체영\n'
 '상을 얻는 영상진단법# 제4조 (보험금을 지급하지 않는 사유)① 회사는 아래의 사유로 보험금 지급사유가 발생한 때에는 보험금을 지급하지 '
 '않습니다.\n'
 '1. 계약자 및 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실\n'
 '2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 이들과 유사한 사태- 3. 지진, 분화, 홍수, 해일 또는 이와 '
 '비슷한 천재지변'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000634',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
