from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 제1항 제1호에서 지정한 특정신체부위에 발생한 질병의 합병증으로 인하여 특정신 체부위 이외의 부위에 발생한 질병으로 보험계약에서 '
 '정한 보험금의 지급사유 또는 보험료 납입면제사유가 발생한 경우 2. 제1항 제2호에서 지정한 특정질병의 합병증으로 인하여 발생한 특정질병 '
 '이외의 질 병으로 보험계약에서 정한 보험금의 지급사유 또는 보험료 납입면제사유가 발생한 경우 3. 상해를 직접적인 원인으로 하여 '
 '보험계약에서 정한 보험금의 지급사유 또는 보험료 납입면제사유가 발생한 경우'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000657',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
