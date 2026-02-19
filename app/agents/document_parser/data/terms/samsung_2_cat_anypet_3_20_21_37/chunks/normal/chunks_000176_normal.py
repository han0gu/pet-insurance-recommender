from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 제1항에도 불구하고 「국가유공자 등 예우 및 지원에 관한 법률」 에 따른 상이자의 증명을 받은 사람 또는 「장애인복지법」 에 따른 '
 '장애인등록증을 발급받은 사람에 대해서는 해당 증명서·장애 인등록증의 사본이나 그 밖의 장애 사실을 증명하는 서류를 제출하는 경우에는 제 '
 '1항의 장애인증 명서는 제출하지 않을 수 있습니다. ③ 장애인으로서 그 장애기간이 기재된 장애인증명서를 제1항 따라 회사에 제출한 때에는 '
 '그 장애기 간 동안은 이를 다시 제출하지 않을 수 있습니다'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 36},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000176',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
