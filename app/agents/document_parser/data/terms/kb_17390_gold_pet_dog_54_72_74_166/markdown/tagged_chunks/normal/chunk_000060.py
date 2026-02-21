from langchain_core.documents import Document

chunk = Document(
    page_content=('|  |\n'
 '| --- |\n'
 '| 관 련 법 규 의료법 ∙ 의료법 제3조(의료기관) 내지 제3조의2(병원등)의 규정에 의한 병원 공 30개 이상의 병상(또는 '
 '요양병상)을 갖춘 병원, 치과병원, 한방병원(또는 요양병 통 원) 사항 ∙ 의료법 제3조(의료기관) 내지 제3조의3(종합병원)의 규정에 '
 '의한 종합병원 100개 이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 진료 |\n'
 '과목마다 전속하는 전문의를 둘 것제15조(상해보험계약 후 알릴 의무) 통약'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000060',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
