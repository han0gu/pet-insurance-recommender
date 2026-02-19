from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 강제집행이란 국가의 집행기관이 채권자를 위하여 집 행권원에 표시된 사법상의 청구권을 국가권력으로 강 제적으로 실현시키는 것을 '
 '말합니다. 2. 담보권실행이란 담보권을 설정한 채권자가 채무를 이 행하지 않은 채무자에 대하여 해당 담보권을 실행하 는 것을 말합니다. '
 '3. 국세 및 지방세 체납처분 절차란 국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하여 체납 된 세금에 대하여 가산금 '
 '징수, 독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 75},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000142',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
