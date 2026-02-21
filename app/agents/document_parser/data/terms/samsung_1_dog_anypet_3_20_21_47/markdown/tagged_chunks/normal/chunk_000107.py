from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기 위하여 지출한 비용\n'
 '제2조(보상하지 않는 손해)회사는 아래의 사유로 인한 손해는 보상하여 드리지 않습니다.- 1. 계약자, 피보험자 또는 이들의 법정대리인의 '
 '고의로 생긴 손해에 대한 배상책임\n'
 '- 2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의 기타 이들과 유사한 사태로 생긴 손해에 대\n'
 '- 한 배상책임\n'
 '- 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변으로 생긴 손해에 대한 배상책임\n'
 '- 4. 피보험자가 소유, 사용 또는 관리하는 재물이 손해를 입었을 경우에 그 재물에 대하여 정당한'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
