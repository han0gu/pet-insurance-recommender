from langchain_core.documents import Document

chunk = Document(
    page_content=('| 8) 추간판탈출증으로 인한 뚜렷한 신경 장해 | 15 |\n'
 '| 9) 추간판탈출증으로 인한 약간의 신경 장해 | 10 |\n'
 '# 나.# 장해판정기준- 1) 척추(등뼈)는 경추에서 흉추, 요추, 제1천추까지를 동일한 부위로 한다.\n'
 '- 제2천추 이하의 천골 및 미골은 체간골의 장해로 평가한다.\n'
 '- 2) 척추(등뼈)의 기형장해는 척추체(척추뼈 몸통을 말하며, 횡돌기 및 극돌\n'
 '- 기는 제외한다. 이하 이 신체부위에서 같다)의 압박률 또는 척추체(척추\n'
 '- 뼈 몸통)의 만곡 정도에 따라 평가한다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000874',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
