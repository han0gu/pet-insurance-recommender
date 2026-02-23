from langchain_core.documents import Document

chunk = Document(
    page_content=('내용을 적용합니다.<br>\uf000 제1항에도 불구하고 다음 각 호 중 어느 한 가지에 해당<br>되는 경우에는 회사는 객관적이고 '
 '합리적인 범위내에서 기<br>존 계약내용에 상응하는 새로운 보장내용으로 계약내용을<br>변경할 수 있습니다.</p><br><p '
 "id='46' data-category='list' style='font-size:16px'>① 관련 법률의 개정 또는 폐지 등에 따라 "
 '약관에서 정한<br>보험금 지급사유 판정기준이 변경되는 경우<br>② 관련 법률의 개정 또는 폐지 등에 따라 약관에서 정한<br>보험금 '
 '지급사유의'),
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
 'indexing': {'chunk_id': 'chunk_000249',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
