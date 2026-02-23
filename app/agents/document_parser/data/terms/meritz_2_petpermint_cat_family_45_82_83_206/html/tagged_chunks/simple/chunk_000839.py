from langchain_core.documents import Document

chunk = Document(
    page_content=('수술비, 진단비 등 보장 범위의 변경 또<br>는 확대는 해당하지 않습니다)된 경우 이를 적용하지 아니<br>할 수 '
 '있습니다.<br>\uf000 제1항의 규정에도 불구하고 다음 중 어느 하나의 사유로<br>보험계약에서 정한 보험금의 지급사유가 발생한 '
 "경우 회사<br>는 보험금을 지급하여 드립니다.</p><br><p id='1' data-category='list' "
 "style='font-size:20px'>① 제1항에서 지정한 특정질병의 합병증으로 인하여 진단<br>확정된 특정질병 이외의 질병으로 "
 '계약에서 정한 보험<br>금의 지급사유가'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000839',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
