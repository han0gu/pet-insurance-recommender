from langchain_core.documents import Document

chunk = Document(
    page_content=("말한다.</p><br><p id='21' data-category='paragraph' style='font-size:16px'>5) "
 '순음청력검사를 실시하기 곤란하거나(청력의 감소가 의<br>심되지만 의사소통이 되지 않는 경우, 만 3세 미만의<br>소아 포함) '
 '검사결과에 대한 검증이 필요한 경우에는<br>“언어청력검사, 임피던스 청력검사, 청성뇌간반응검<br>사(ABR), 이음향방사검사”등을 '
 "추가실시 후 장해를<br>평가한다.</p><h1 id='22' style='font-size:20px'>다"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000936',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
