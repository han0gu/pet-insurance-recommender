from langchain_core.documents import Document

chunk = Document(
    page_content=('보험계약일 보장개시일(책임개시일)\n'
 '◄───── 30일 ─────►\n'
 '2022년 8월 1일 2022년 8월 31일# 제2조 (보험금을 지급하지 않는 사유)회사는 아래의 사유를 원인으로 하여 생긴 손해는 '
 '보상하지 않습니다.- 1. 계약자 및 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실\n'
 '- 2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 이들과 유사한 사태\n'
 '- 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000466',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
