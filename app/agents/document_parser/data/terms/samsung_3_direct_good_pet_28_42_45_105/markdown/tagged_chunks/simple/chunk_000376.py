from langchain_core.documents import Document

chunk = Document(
    page_content=('조상으로부터 직계로 내려와 자기에 이르는 사이의 혈족. 부모, 조부모 등\n'
 '[방계혈족]\n'
 '자기의 형제자매와 형제자매의 직계비속, 직계존속의 형제자매 및 그 형제자매의 직계비속④ 보험수익자는 통지를 받은 날(제3항에 따라 '
 '계약자에게 통지된 경우에는 계약자가 통\n'
 '지를 받은 날을 말합니다)부터 15일 이내에 제1항의 절차를 이행할 수 있습니다.# 제24조 (계약자의 임의해지)계약자는 특별약관이 '
 '소멸하기 전에는 언제든지 이 특별약관을 해지할 수 있으며, 이 경'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000376',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
