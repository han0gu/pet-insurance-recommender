from langchain_core.documents import Document

chunk = Document(
    page_content=('통지된 경우에는 계약자가 통지를 받은 날을 말합니다)부터\n'
 '15일 이내에 제1항의 절차를 이행할 수 있습니다.# 【용어풀이】1. 강제집행이란 국가의 집행기관이 채권자를 위하여 집\n'
 '행권원에 표시된 사법상의 청구권을 국가권력으로 강\n'
 '제적으로 실현시키는 것을 말합니다.\n'
 '2. 담보권실행이란 담보권을 설정한 채권자가 채무를 이\n'
 '행하지 않은 채무자에 대하여 해당 담보권을 실행하\n'
 '는 것을 말합니다.\n'
 '3. 국세 및 지방세 체납처분 절차란 국세 또는 지방세를\n'
 '체납할 경우 국세 기본법 및 지방세법에 의하여 체납'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000113',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
