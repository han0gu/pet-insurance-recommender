from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의 움직임을 식별할 수 있을 정도의 시력상태\n'
 '- 주2) 안전수지 : 시표의 가장 큰 글씨를 읽을 수 있는 정도의 시력은 아\n'
 '- 니나 눈 앞 30cm 이내에서 손가락의 개수를 식별할 수 있을 정도의\n'
 '- 시력상태\n'
 '- 5) 안구(눈동자) 운동장해의 판정은 질병의 진단 또는 외상 후 1년 이상이\n'
 '- 지난 뒤 그 장해 정도를 평가한다.\n'
 '- 6) ‘안구(눈동자)의 뚜렷한 운동장해’ 라 함은 아래의 두 경우 중 하나에\n'
 '- 해당하는 경우를 말한다.\n'
 '가) 한 눈의 안구(눈동자)의 주시야(머리를 움직이지 않고 눈만을 움직'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000841',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
