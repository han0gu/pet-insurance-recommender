from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 질병과 부상의 주<br>증상과 합병증상 및 이에 대한 치료를 받는 과정에서<br>일시적으로 나타나는 증상은 장해에 포함되지 '
 '않는다.<br>2) “영구적”이라 함은 원칙적으로 치유하는 때 장래 회<br>복할 가망이 없는 상태로서 정신적 또는 육체적 '
 '훼손<br>상태임이 의학적으로 인정되는 경우를 말한다.<br>3) “치유된 후”라 함은 상해 또는 질병에 대한 치료의<br>효과를 기대할 '
 '수 없게 되고 또한 그 증상이 고정된<br>상태를 말한다.<br>4) 다만, 영구히 고정된 증상은 아니지만 치료종결후 한시<br>적으로'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000907',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
