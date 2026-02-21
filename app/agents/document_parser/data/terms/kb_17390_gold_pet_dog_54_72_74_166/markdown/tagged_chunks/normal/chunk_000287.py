from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항의 "성형수술" 은 피보험자가 사고발생시점에 만15세 미만일 경우 부득이\n'
 '- 78 -사고일로부터 2년이 지난 후에 성형수술이 가능하다는 진단을 받은 경우에는 그# 진단으로 대체할 수 있습니다.- 제2조(보험금 '
 '지급에 관한 세부규정)\n'
 '- \uf000 제1조(보험금의 지급사유)의 상해흉터복원수술비Ⅱ(안면부)는 하나의 사고로 동\n'
 '- 일부위에 대한 성형수술을 2회 이상 받은 경우에는 최초로 받은 수술에 대해서만\n'
 '- 지급합니다.\n'
 '- \uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000287',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
