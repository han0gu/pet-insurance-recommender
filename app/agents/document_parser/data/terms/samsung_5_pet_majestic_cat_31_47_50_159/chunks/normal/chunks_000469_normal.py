from langchain_core.documents import Document

chunk = Document(
    page_content=('제 2조 (보험금 지급에 관한 세부규정)\n'
 '① 제1조(보험금의 지급사유)의‘사망’에는 보험기간에 다음 어느 하나의 사유가 발생 한 경우를 포함합니다.\n'
 '1. 실종선고를 받은 경우: 법원에서 인정한 실종기간이 끝나는 때에 사망한 것으로 봅 니다. 2. 관공서에서 수해, 화재나 그 밖의 '
 '재난을 조사하고 사망한 것으로 통보하는 경우: 가족관계등록부에 기재된 사망연월일을 기준으로 합니다.\n'
 '<용어풀이>\n'
 '[실종선고]\n'
 '어떤 사람의 생사불명의 상태가 일정기간 이상 계속될 때 이해관계가 있는 사람의 청구에 의해 사 망한 것으로 간주하는 법원의 결정'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 86},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000469',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
