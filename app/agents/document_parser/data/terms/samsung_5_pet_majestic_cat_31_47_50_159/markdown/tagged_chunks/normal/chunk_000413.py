from langchain_core.documents import Document

chunk = Document(
    page_content=('입금액을 보험금 지급사유 발생일(단, 해당월에 보험금 지급사유 발생일이 없는 경우에는\n'
 '해당월의 마지막 날로 합니다)에 반려동물 양육자금Ⅱ으로 보험수익자에게 지급합니다.# 제 2조 (보험금 지급에 관한 세부규정)① '
 '「호스피스∙완화의료 및 임종과정에 있는 환자의 연명의료결정에 관한 법률」에 따른\n'
 '연명의료중단 등 결정 및 그 이행으로 피보험자가 사망하는 경우 연명의료중단 등 결\n'
 "정 및 그 이행은 제1조(보험금의 지급사유) '사망'의 원인 및 '사망보험금' 지급에 영향\n"
 '을 미치지 않습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000413',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
