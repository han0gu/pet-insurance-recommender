from langchain_core.documents import Document

chunk = Document(
    page_content=('【수의사법 제2조(정의)】\n'
 '이 법에서 사용하는 용어의 뜻은 다음과 같다.\n'
 '1. "수의사"란 수의업무를 담당하는 사람으로서 농림축 산식품부장관의 면허를 받은 사람을 말한다. 4. "동물병원"이란 동물진료업을 하는 '
 '장소로서 제17조 에 따른 신고를 한 진료기관을 말한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 156},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000533',
              'chunk_char_len': 149,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
