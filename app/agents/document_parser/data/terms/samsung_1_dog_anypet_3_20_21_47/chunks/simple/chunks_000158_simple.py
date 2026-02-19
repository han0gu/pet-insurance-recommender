from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자, 피보험자 또는 이들의 가족 또는 사용인의 고의 또는 중대한 과실 2. 보험개시일로부터 그 날을 포함하여 30일 이내에 '
 '발생한 손해. 단, 이 반려동물 사망위로금 특 별약관을 갱신하는 경우에는 적용하지 않습니다. 3. 반려동물을 범죄행위, 경주, 수색, '
 '폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적으로 이용함 으로써 발생한 손해 4. 수의사의 치료상의 과오로 생긴 손해, 수의사 '
 '자격이 없는 자의 치료행위로 인한 손해 5. 지진, 분화, 해일, 홍수 또는 이와 비슷한 천재지변 6'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 30},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000158',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
