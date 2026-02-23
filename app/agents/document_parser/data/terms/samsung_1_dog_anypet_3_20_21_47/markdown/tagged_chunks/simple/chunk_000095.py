from langchain_core.documents import Document

chunk = Document(
    page_content=("- 발생하여 그 치료를 직접적인 목적으로 국내에서 수의사에게 수술(이하 '사고'라 합니다)을 받은 경\n"
 '- 우 수술 당일 발생한 수술비 및 치료비를 보통약관에서 보상하는 치료비보험금에 추가하여 보상하\n'
 '- 여 드립니다.\n'
 '- ② 제1항의 수술비용 확대보장에 대한 회사의 보장은 보험개시일로부터 30일 이내(이하"대기기간")에\n'
 '- 발생한 질병(단, 암, 백내장, 녹내장, 심장질환, 신장질환, 방광질환 및 각종결석의 대기기간은 90\n'
 '- 일 적용)으로 인한 손해는 보상하여 드리지 않습니다. 단, 이 수술비용 확대보장 특별약관을 갱신'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'eye', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000095',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
