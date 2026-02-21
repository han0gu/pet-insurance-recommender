from langchain_core.documents import Document

chunk = Document(
    page_content=('한 경우를 포함합니다.- 1. 실종선고를 받은 경우: 법원에서 인정한 실종기간이 끝나는 때에 사망한 것으로 봅\n'
 '- 니다.\n'
 '- 2. 관공서에서 수해, 화재나 그 밖의 재난을 조사하고 사망한 것으로 통보하는 경우:\n'
 '- 가족관계등록부에 기재된 사망연월일을 기준으로 합니다.\n'
 '# <용어풀이># [실종선고]어떤 사람의 생사불명의 상태가 일정기간 이상 계속될 때 이해관계가 있는 사람의 청구에 의해 사\n'
 '망한 것으로 간주하는 법원의 결정② 「호스피스·완화의료 및 임종과정에 있는 환자의 연명의료결정에 관한 법률」에 따'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000396',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
