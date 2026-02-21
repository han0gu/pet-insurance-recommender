from langchain_core.documents import Document

chunk = Document(
    page_content=('급의료기관) 또는 응급의료에 관한 법률 제35조의2(응급의료기관 외의 의료기관)에서\n'
 '정하는 시장 · 군수 · 구청장에게 신고되고 그 신고가 수리된 응급의료시설을 말합니\n'
 '다.\n'
 '② 관련 법령이 개정된 경우 개정된 내용을 적용하며, 의료기관의 「응급실」 해당여부는\n'
 '내원 당시 기준으로 판단합니다.- \n'
 '# 제5조 (보험금을 지급하지 않는 사유)① 회사는 다음 중 어느 한 가지로 보험금 지급사유가 발생한 때에는 보험금을 지급하지\n'
 '않습니다.- 1. 피보험자가 고의로 자신을 해친 경우. 다만, 피보험자가 심신상실 등으로 자유로운'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000315',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
