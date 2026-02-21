from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>【직업】</h1><br><p id='26' data-category='list' "
 "style='font-size:20px'>1) 생계유지 등을 위하여 일정한 기간동안(예: 6개월 이<br>상) 계속하여 종사하는 "
 '일<br>2) 1)에 해당하지 않는 경우에는 개인의 사회적 신분에<br>따르는 위치나 자리를 말함<br>예) 학생, 미취학아동, 무직 '
 "등</p><h1 id='27' style='font-size:18px'>【직무】</h1><br><p id='28'"),
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
 'indexing': {'chunk_id': 'chunk_000086',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
