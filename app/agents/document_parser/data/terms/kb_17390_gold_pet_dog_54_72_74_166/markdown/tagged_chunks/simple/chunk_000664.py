from langchain_core.documents import Document

chunk = Document(
    page_content=('배상책임에 있어 회사와 계약자간에 약정한 금액으로 피보 특\n'
 '∙ 핵연료물질에 의하여 오염된 물질 : 원자핵 분열 생성물을 포함합니다. 별\n'
 '험자가 법률상의 배상책임을 부담함으로써 입은 손해 중 보\n'
 '보상한도액 약\n'
 '험금 등의 지급한도에 따라 회사가 책임지는 금액의 최대 \uf000 회사는 그 원인의 직접, 간접을 묻지 않고 아래의 사유로 인한 손해는 '
 '보상하여\n'
 '관\n'
 '한도를 말합니다. 드리지 않습니다.\n'
 '1. 피보험자의 피용인이 피보험자의 업무에 종사 중에 입은 신체의 피해로 인한\n'
 '배상책임\n'
 '제3조(보상하는 손해)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000664',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
