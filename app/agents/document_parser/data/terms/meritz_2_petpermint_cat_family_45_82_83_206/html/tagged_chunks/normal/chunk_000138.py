from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>\uf000 회사가 제1항에 따라 제공될 약관 및 계약자 보관용 청<br>약서를 청약할 때 "
 '계약자에게 전달하지 않거나 약관의 중요<br>한 내용을 설명하지 않은 때 또는 계약을 체결할 때 계약자<br>가 청약서에 자필서명을 하지 '
 "않은 때에는 계약자는 계약이<br>성립한 날부터 3개월 이내에 계약을 취소할 수 있습니다.</p><br><h1 id='94' "
 "style='font-size:20px'>【자필서명】</h1><br><p id='95' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000138',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
