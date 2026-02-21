from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 장애인전용보험·공제로 표시된 보험·공제의 보험료·공제료를 말한다.\n'
 '- ② 법 제59조의4 제1항 제2호에서 "대통령령으로 정하는 보험료"란 다음 각 호의 어느\n'
 '- 하나에 해당하는 보험·보증·공제의 보험료·보증료·공제료 중 기획재정부령으로 정하\n'
 '- 는 것을 말한다.\n'
 '- 1. 생명보험\n'
 '- 2. 상해보험\n'
 '- 3. 화재·도난이나 그 밖의 손해를 담보하는 가계에 관한 손해보험\n'
 '- 4.「수산업협동조합법」,「신용협동조합법」또는「새마을금고법」에 따른 공제\n'
 '- 5.「군인공제회법」,「한국교직원공제회법」,「대한지방행정공제회법」,「경찰공제회'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000204',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
