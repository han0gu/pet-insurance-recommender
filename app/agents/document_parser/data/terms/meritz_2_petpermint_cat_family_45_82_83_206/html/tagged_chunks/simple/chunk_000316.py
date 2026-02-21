from langchain_core.documents import Document

chunk = Document(
    page_content=("id='53' style='font-size:20px'>제8조(계약 후 알릴 의무)</h1><br><p id='54' "
 "data-category='paragraph' style='font-size:20px'>\uf000 계약자 또는 피보험자는 보험기간 중에 "
 '다음 각 호의 변<br>경이 발생한 경우에는 우편, 전화, 방문 등의 방법으로 지<br>체없이 회사에 알려야 합니다.</p><br><p '
 "id='55' data-category='list' style='font-size:16px'>① 청약서의 기재사항을 변경하고자 할 때 "
 '또는'),
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
 'indexing': {'chunk_id': 'chunk_000316',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
