from langchain_core.documents import Document

chunk = Document(
    page_content=('경우 “<붙임>일상생활 기본동작(ADLs) 제<br>한 장해평가표”상 지급률이 10% 미만인 경우에는<br>보장대상이 되는 장해로 '
 '인정하지 않는다.<br>다) 신경계의 장해로 발생하는 다른 신체부위의 장해<br>(눈, 귀, 코, 팔, 다리 등)는 해당 장해로도 '
 '평가<br>하고 그 중 높은 지급률을 적용한다.<br>라) 뇌졸중, 뇌손상, 척수 및 신경계의 질환 등은 발<br>병 또는 외상 후 '
 '12개월 동안 지속적으로 치료한<br>후에 장해를 평가한다.<br>그러나, 12개월이 지났다고 하더라도 뚜렷하게<br>기능 향상이 '
 '진행되고'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_001089',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
