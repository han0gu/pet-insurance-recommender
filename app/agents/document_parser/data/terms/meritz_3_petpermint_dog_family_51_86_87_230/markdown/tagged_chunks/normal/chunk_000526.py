from langchain_core.documents import Document

chunk = Document(
    page_content=('186# 부담하지 않습니다.㉤ 피보험자가「배상책임 관련 특별약관 일반조항」제9\n'
 '조(손해배상청구에 대한 회사의 해결)의 제2항 및\n'
 '제3항의 회사의 요구에 따르기 위하여 지출한 비용\uf000 제4항의 손해에 대하여 다음과 같이 보상합니다.- ① 제4항 제1호의 '
 '손해배상금 : 보험가입금액 한도로 보\n'
 '- 상하되, 매회의 사고마다 자기부담금 3만원을 초과하\n'
 '- 는 경우에 한하여 그 초과하는 배상책임 손해액을 보\n'
 '- 상합니다.\n'
 '- ② 제4항 제2호 ㉠, ㉡ 또는 ㉤의 비용 : 비용의 전액을\n'
 '- 보상합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000526',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
