from langchain_core.documents import Document

chunk = Document(
    page_content=('말합니다.1. 보험증권에 기재된 피보험자(이하「피보험자 본인」이라 합니다)\n'
 '2. 피보험자 본인의 가족관계등록상 또는 주민등록상에 기재된 배우자(이하「배우자」\n'
 '라 합니다)\n'
 '3. 피보험자 본인 또는 배우자와 생계를 같이 하는 동거 친족 및 별거 중인 미혼자녀② 위 제1항에서 피보험자 본인과 본인 이외의 '
 '피보험자와의 관계는 사고발생 당시의 관\n'
 '계를 말합니다.제2관 보험금의 지급- 2 -제4조(보험금의 지급사유)① 회사는 보험기간 중에 보험증권에 기재된 반려동물에게 질병 또는 '
 '상해가 발생하여 그'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000009',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
