from langchain_core.documents import Document

chunk = Document(
    page_content=('- 별약관을 갱신하는 경우에는 적용하지 않습니다.\n'
 '- 3. 반려동물을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적으로 이용함\n'
 '- 으로써 발생한 손해\n'
 '- 4. 수의사의 치료상의 과오로 생긴 손해, 수의사 자격이 없는 자의 치료행위로 인한 손해\n'
 '- 5. 지진, 분화, 해일, 홍수 또는 이와 비슷한 천재지변\n'
 '- 6. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소요, 기타 이들과 유사한 사태\n'
 '- 7. 핵연료물질(사용이 끝난 연료를 포함합니다. 이하 같습니다) 또는 핵연료 물질에 의하여 오염된 물'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000129',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
