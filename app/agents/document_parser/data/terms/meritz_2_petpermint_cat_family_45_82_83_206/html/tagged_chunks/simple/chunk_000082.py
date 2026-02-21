from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 보험자가 계약당시에<br>그 사실을 알았거나 중대한 과실로 인하여 알지 못한 때<br>에는 그러하지 아니하다.</p><h1 '
 "id='17' style='font-size:20px'>【상법 제651조의2(서면에 의한 질문의 효력)】</h1><br><p "
 "id='18' data-category='paragraph' style='font-size:20px'>보험자가 서면으로 질문한 사항은 "
 "중요한 사항으로 추정<br>한다.</p><h1 id='19' style='font-size:20px'>【사례】</h1><br><p "
 "id='20'"),
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
 'indexing': {'chunk_id': 'chunk_000082',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
