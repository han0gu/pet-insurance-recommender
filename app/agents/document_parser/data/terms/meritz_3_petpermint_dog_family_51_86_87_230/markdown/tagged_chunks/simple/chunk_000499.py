from langchain_core.documents import Document

chunk = Document(
    page_content=('여 회사의 비용으로 이를 해결할 수 있습니다. 이 경우 회\n'
 '사의 요구가 있으면 계약자 및 피보험자는 이에 협력하여야\n'
 '합니다.\n'
 '\uf000 계약자 및 피보험자가 정당한 이유없이 제2항 및 제3항\n'
 '의 요구에 협조하지 않은 때에는 회사는 그로 인하여 늘어\n'
 '난 손해는 보상하지 않습니다.179【손해배상청구에 대한 회사의 해결 예시】![image](/image/placeholder)\n'
 '① 손해배상책임사고발생 ② 보험금 지급청구\n'
 '피보험자 피해자 보험사\n'
 '③ 피보험자를 대신하여\n'
 '항변으로써 대항가능※ 항변이란 어떤 일을 부당하다고 여겨 따지거나 반대하'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000499',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
