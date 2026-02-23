from langchain_core.documents import Document

chunk = Document(
    page_content=('이 법에서 의료기관이라 함은 의료인이 공중 또는 특수 다수인을 위하여 의료・\n'
 '조산의 업을 행하는 곳을 말합니다. 의료기관의 종별은 종합병원・병원・치과병\n'
 '원・한방병원・요양병원・정신병원・의원・치과의원・한의원 및 조산원으로 나# 누어집니다.제5조(보험금의 분담)\n'
 '\uf000 회사는 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약\n'
 '을 포함합니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각\n'
 '각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 손해를보상합니다.\n'
 '다른 계약이 없을 때'),
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
 'indexing': {'chunk_id': 'chunk_000726',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
