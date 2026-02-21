from langchain_core.documents import Document

chunk = Document(
    page_content=('잘라내었을 때<br>나) 방광 기능상실로 영구적인 요도루, 방광루, 요관<br>장문합 상태<br>다) 위, 췌장을 50% 이상 잘라내었을 '
 "때<br>라) 대장절제, 항문 괄약근 등의 기능장해로 영구적으</p><footer id='24' "
 "style='font-size:14px'>199</footer><p id='25' data-category='paragraph' "
 "style='font-size:20px'>로 장루, 인공항문을 설치한 경우(치료과정에서<br>일시적으로 발생하는 경우는 "
 "제외)</p><br><p id='26'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001077',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
