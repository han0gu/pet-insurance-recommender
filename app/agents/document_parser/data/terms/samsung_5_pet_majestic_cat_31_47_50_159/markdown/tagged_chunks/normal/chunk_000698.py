from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보험료 납입면제사유가 발생한 경우\n'
 '- 2. 제1항 제2호에서 지정한 특정질병의 합병증으로 인하여 발생한 특정질병 이외의 질\n'
 '- 병으로 보험계약에서 정한 보험금의 지급사유 또는 보험료 납입면제사유가 발생한\n'
 '- 경우\n'
 '- 3. 상해를 직접적인 원인으로 하여 보험계약에서 정한 보험금의 지급사유 또는 보험료\n'
 '- 납입면제사유가 발생한 경우\n'
 '- ⑦ 피보험자가 회사가 정한 회사가 보험금을 지급하지 않는 기간의 종료일을 포함하여\n'
 '- 계속하여 입원한 경우 그 입원에 대해서는 회사가 보험금을 지급하지 않는 기간 종료'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000698',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
