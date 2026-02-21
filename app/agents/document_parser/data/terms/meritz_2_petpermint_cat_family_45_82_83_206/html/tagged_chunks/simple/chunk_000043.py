from langchain_core.documents import Document

chunk = Document(
    page_content=("id='60' data-category='paragraph' style='font-size:16px'>기간이 제1항의 지급기일을 초과할 "
 '것이 명백히 예상되는<br>경우에는 그 구체적인 사유와 지급예정일 및 보험금 가지급<br>제도(회사가 추정하는 보험금의 50% 이내를 '
 '지급)에 대하여<br>피보험자 또는 보험수익자에게 즉시 통지합니다'),
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
 'indexing': {'chunk_id': 'chunk_000043',
              'chunk_char_len': 187,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
