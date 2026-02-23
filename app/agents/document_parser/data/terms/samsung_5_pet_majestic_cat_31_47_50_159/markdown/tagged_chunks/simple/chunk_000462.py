from langchain_core.documents import Document

chunk = Document(
    page_content=('- 상 동거중인 동거 친족(민법 제 777조)\n'
 '- 4. 피보험자 본인 또는 배우자와 생계를 같이하는 별거 중인 미혼자녀\n'
 '<관련법규>[민법 제777조(친족의 범위)에서 규정한 친족의 범위]: 8촌 이내의 혈족, 4촌 이내의 인척, 배우자# 제6조 (보험금을 '
 '지급하지 않는 사유)① 회사는 아래의 사유로 보험금 지급사유가 발생한 때에는 보험금을 지급하지 않습니다.- 1. 계약자 및 피보험자, '
 '이들의 가족 또는 사용인의 고의 또는 중대한 과실\n'
 '- 2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 이들과 유사한 사태'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000462',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
