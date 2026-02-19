from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 8촌이내의 혈족 2. 4촌이내의 인척 3. 배우자\n'
 '\uf000 제2항에서 피보험자 본인과 본인 이외의 피보험자와의 관계는 사고발생 당시의 관계를 말합니다.\n'
 '\uf000 1사고당 보상하는 손해의 범위는 아래와 같습니다.\n'
 '① 피보험자가 피해자에게 지급할 책임을 지는 법률상의 손해배상금 ② 계약자 또는 피보험자가 지출한 아래의 비용'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 186},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000632',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
