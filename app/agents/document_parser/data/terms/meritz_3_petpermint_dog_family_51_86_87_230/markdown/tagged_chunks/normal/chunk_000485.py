from langchain_core.documents import Document

chunk = Document(
    page_content=('| 보험료 | 계약에서 정한 손해를 보장하는데 필요한 보험 료를 말합니다. |\n'
 '# 제3조(손해의 발생과 통지)\uf000 계약자 또는 피보험자는 아래와 같은 사실이 있는 경우\n'
 '에는 지체없이 그 내용을 회사에 알려야 합니다.- ① 사고가 발생하였을 경우 사고발생의 때와 곳, 피해자\n'
 '- 의 주소와 성명, 사고상황 및 이들 사항의 증인이 있\n'
 '- 을 경우 그 주소와 성명\n'
 '- ② 피해자로부터 손해배상청구를 받았을 경우\n'
 '- ③ 피해자로부터 손해배상책임에 관한 소송을 제기 받았\n'
 '- 을 경우\n'
 '\uf000 계약자 또는 피보험자가 제1항 각 호의 통지를 게을리하'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000485',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
