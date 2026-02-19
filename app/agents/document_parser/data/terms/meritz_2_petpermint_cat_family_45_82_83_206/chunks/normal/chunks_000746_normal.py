from langchain_core.documents import Document

chunk = Document(
    page_content=('나) 치매의 장해평가는 임상적인 증상 뿐 아니라 뇌영 상검사(CT 및 MRI, SPECT 등)를 기초로 진단되어 져야 하며, 18개월 '
 '이상 지속적인 치료 후 평가한 다. 다만, 진단시점에 이미 극심한 치매 또는 심 한 치매로 진행된 경우에는 6개월간 지속적인 치 료 후 '
 '평가한다. 다) 치매의 장해평가는 전문의(정신건강의학과, 신경 과)에 의한 임상치매척도(한국판 Expanded Clinical '
 'Dementia Rating) 검사결과에 따른다.\n'
 '4) 뇌전증'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 203},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000746',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
