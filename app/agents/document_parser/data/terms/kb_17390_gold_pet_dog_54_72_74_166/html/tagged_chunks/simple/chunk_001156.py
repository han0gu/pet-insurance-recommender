from langchain_core.documents import Document

chunk = Document(
    page_content=('. 려동<br>제5조(보상하지 않는 손해)<br>1. 사고가 발생하였을 경우 사고가 발생한 때와 곳, 피해자의 주소와 성명, 사고 '
 '물<br>\uf000 회사는 아래의 사유로 인한 손해는 보상하여 드리지 않습니다.<br>상황 및 이들 사항의 증인이 있을 경우 그 주소와 '
 '성명<br>1. 계약자, 피보험자 또는 이들의 법정대리인의 고의로 생긴 손해에 대한 배상책임<br>2. 피해자로부터 손해배상청구를 받았을 '
 '경우<br>2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 이들과 유사한 사태<br>3'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001156',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
