from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:18px'>【직무】</h1><br><p id='28' data-category='paragraph' "
 "style='font-size:20px'>직책이나 직업상 책임을 지고 담당하여 맡은 일</p><br><p id='29' "
 "data-category='list' style='font-size:18px'>② 보험증권에 기재된 피보험자의 운전 목적이 변경된 "
 '경우<br>예) 자가용에서 영업용으로 변경, 영업용에서 자가용<br>으로 변경 등<br>③ 보험증권에 기재된 피보험자의 운전여부가 변경된 '
 '경우<br>예)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000087',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
